import EduNets: update!, model2vector!, model2vector, project!, forward!, gradient!, fgradient!;
import EduNets;

export Model, update!, model2vector!, model2vector, project!, forward!, fgradient!, addsoftmax;

type ModelCache{A<:AbstractFloat}
	partOutput::StridedMatrix{A};
end

function ModelCache(; T::DataType = Float32)::ModelCache
	return ModelCache(StridedMatrix{T}(0));
end

type Model{A<:Tuple, B<:Tuple, C<:Tuple, D<:Tuple}<:EduNets.AbstractModel
	domainModel::A;
	pathModel::B;
	queryModel::C;
	urlModel::D;

	cache::ModelCache;
end

function Model(domainModel::Tuple, pathModel::Tuple, queryModel::Tuple, urlModel::Tuple; T::DataType = Float32)
	return Model(domainModel, pathModel, queryModel, urlModel, ModelCache(; T = T));
end

# update = vector2model
function update!(model::Model, theta::Vector; offset::Int = 1)
	offset = update!(model.domainModel, theta; offset = offset);
	offset = update!(model.pathModel, theta; offset = offset);
	offset = update!(model.queryModel, theta; offset = offset);
	offset = update!(model.urlModel, theta; offset = offset);
end

function model2vector!(model::Model, theta::Vector; offset::Int = 1)
	offset = model2vector!(model.domainModel, theta; offset = offset);
	offset = model2vector!(model.pathModel, theta; offset = offset);
	offset = model2vector!(model.queryModel, theta; offset = offset);
	offset = model2vector!(model.urlModel, theta; offset = offset);
end

function model2vector(model::Model)
	return vcat(model2vector(model.domainModel), model2vector(model.pathModel), model2vector(model.queryModel), model2vector(model.urlModel));
end

function project!(model::Model, dataset::Dataset)
	od = forward!(model.domainModel, dataset.domains.x, (dataset.domains.bags,));
	op = forward!(model.pathModel, dataset.paths.x, (dataset.paths.bags,));
	oq = forward!(model.queryModel, dataset.queries.x, (dataset.queries.bags,));

	size1 = size(od[end], 1) + size(op[end], 1) + size(oq[end], 1);
	size2 = size(od[end], 2);

	if (size(model.cache.partOutput, 1) < size1 ) || (size(model.cache.partOutput, 2) < size2)
		model.cache.partOutput = StridedMatrix{Float32}(size1, size2);
	end

	dsize = size(od[end], 1);
	psize = size(op[end], 1);

	model.cache.partOutput[1:dsize, 1:size2] = od[end];
	model.cache.partOutput[dsize + 1:dsize + psize, 1:size2] = op[end];
	model.cache.partOutput[dsize + psize + 1:size1, 1:size2] = oq[end];

	oo = forward!(model.urlModel, model.cache.partOutput[1:size1, 1:size2])[end];
	return oo;
end

function forward!(model::Model, dataset::Dataset)
	od = forward!(model.domainModel, dataset.domains.x, (dataset.domains.bags,));
	op = forward!(model.pathModel, dataset.paths.x, (dataset.paths.bags,));
	oq = forward!(model.queryModel, dataset.queries.x, (dataset.queries.bags,));

	size1 = size(od[end], 1) + size(op[end], 1) + size(oq[end], 1);
	size2 = size(od[end], 2);

	if (size(model.cache.partOutput, 1) < size1 ) || (size(model.cache.partOutput, 2) < size2)
		model.cache.partOutput = StridedMatrix{Float32}(size1, size2);
	end

	dsize = size(od[end], 1);
	psize = size(op[end], 1);

	model.cache.partOutput[1:dsize, 1:size2] = od[end];
	model.cache.partOutput[dsize + 1:dsize + psize, 1:size2] = op[end];
	model.cache.partOutput[dsize + psize + 1:size1, 1:size2] = oq[end];

	oo = forward!(model.urlmodel, model.cache.partOutput[1:size1, 1:size2]);
	return oo;
end

function fgradient!(model::Model,loss::EduNets.AbstractLoss, dataset::Dataset, g::Model)
	od = forward!(model.domainModel, dataset.domains.x, (dataset.domains.bags,));
	op = forward!(model.pathModel, dataset.paths.x, (dataset.paths.bags,));
	oq = forward!(model.queryModel, dataset.queries.x, (dataset.queries.bags,));

	size1 = size(od[end], 1) + size(op[end], 1) + size(oq[end], 1);
	size2 = size(od[end], 2);

	if (size(model.cache.partOutput, 1) < size1 ) || (size(model.cache.partOutput, 2) < size2)
		model.cache.partOutput = StridedMatrix{Float32}(size1, size2);
	end

	dsize = size(od[end], 1);
	psize = size(op[end], 1);

	model.cache.partOutput[1:dsize, 1:size2] = od[end];
	model.cache.partOutput[dsize + 1:dsize + psize, 1:size2] = op[end];
	model.cache.partOutput[dsize + psize + 1:end, 1:size2] = oq[end];

	oo = forward!(model.urlmodel, model.cache.partOutput[1:size1, 1:size2]);

	(f, goo) = gradient!(loss, oo[end], dataset.y); #calculate the gradient of the loss function 

	(f1, go) = EduNets.fbackprop!(model.urlModel, oo, goo, g.urlModel);

	dsize = size(model.domainModel[end], 1);
	psize = size(model.pathModel[end], 1);
	qsize = size(model.queryModel[end], 1);

	god = view(go, 1:dsize, :);
	gop = view(go, dsize + 1:dsize + psize, :);
	goq = view(go, dsize + psize + 1:dsize + psize + qsize, :);

	f2 = fgradient!(model.domainModel, od, (dataset.domains.bags,), god, g.domainModel);
	f3 = fgradient!(model.pathModel, op, (dataset.paths.bags,), gop, g.pathModel);
	f4 = fgradient!(model.queryModel, oq, (dataset.queries.bags,), goq, g.queryModel);
	return f + f1 + f2 + f3 + f4;
end

function addsoftmax(model::Model; T::DataType = Float32)
	return Model(model.domainModel, model.pathModel, model.queryModel, (model.urlModel..., EduNets.SoftmaxLayer(size(model.urlModel[end], 2), T = T)));
end
